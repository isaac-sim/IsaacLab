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
  properties, by selecting a physics variant on the generated USD file. Conversion now raises when
  the asset does not offer the requested variant, as happens when requesting ``"physx"`` for a URDF
  whose joints are all fixed.

* Fixed :meth:`~isaaclab.utils.dict.class_to_dict` expanding enum values into their internal
  members, which wrote unusable entries into serialized configurations.

Changed
^^^^^^^

* Changed :func:`~isaaclab.sim.utils.select_usd_variants` to raise when a variant set listed in
  :obj:`~isaaclab.sim.utils.REQUIRED_VARIANT_SETS` is absent from the prim or does not offer the
  requested variant. ``"Physics"`` is the only such set today: USD accepts a selection naming a
  variant that does not exist and composes the prim as if nothing were selected, so the asset
  spawned as plain geometry with no diagnostic. Other variant sets keep the previous behaviour of
  logging a warning and continuing.
